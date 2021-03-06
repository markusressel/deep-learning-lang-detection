<?php namespace app\controllers\Admin;

use Suenos\Categories\CategoryRepository;
use Suenos\Orders\OrderRepository;
use Suenos\Products\ProductRepository;
use Suenos\Users\UserRepository;

class DashboardController extends \BaseController {

    /**
     * @var categoryRepository
     */
    private $categoryRepository;
    /**
     * @var ProductRepository
     */
    private $productRepository;
    /**
     * @var UserRepository
     */
    private $userRepository;
    /**
     * @var OrderRepository
     */
    private $orderRepository;

    function __construct(CategoryRepository $categoryRepository, ProductRepository $productRepository, UserRepository $userRepository, OrderRepository $orderRepository)
    {

        $this->categoryRepository = $categoryRepository;
        $this->productRepository = $productRepository;
        $this->userRepository = $userRepository;
        $this->orderRepository = $orderRepository;
    }

    /**
     * Display a listing of the resource.
     * GET /dashboard
     *
     * @return Response
     */
    public function index()
    {
        $categories = $this->categoryRepository->getLasts();
        $total_categories = $this->categoryRepository->getTotal();
        $products = $this->productRepository->getLasts();
        $total_products = $this->productRepository->getTotal();
        $orders = $this->orderRepository->getLasts();
        $total_orders = $this->orderRepository->getTotal();
        $users = $this->userRepository->getLasts();
        $total_users = $this->userRepository->getTotal();

        return \View::make('admin.dashboard.index')->withCategories($categories)->withTc($total_categories)
            ->withProducts($products)->withTp($total_products)
            ->withOrders($orders)->withTo($total_orders)
            ->withUsers($users)->withTu($total_users);
    }


}